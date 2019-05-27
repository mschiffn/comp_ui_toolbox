%
% compute two-dimensional discrete wavelet transform for various options
%
% author: Martin F. Schiffner
% date: 2016-08-13
% modified: 2019-05-27
%
classdef wavelet < linear_transforms.orthonormal_linear_transform

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        type                                            % type of wavelet: 'Haar', 'Beylkin', 'Coiflet', 'Daubechies', 'Symmlet', 'Vaidyanathan', 'Battle'
        parameter ( 1, 1 ) { mustBeInteger } = 20       % type-dependent parameter related to the support and vanishing moments of the wavelets
        N_dimensions ( 1, 1 ) { mustBePositive, mustBeInteger } = 2         % number of points ( dyadic )
        scale_finest ( 1, 1 ) { mustBeNonnegative, mustBeInteger } = 9      % finest scale ( fine level )
        scale_coarsest ( 1, 1 ) { mustBeNonnegative, mustBeInteger } = 0	% coarsest scale ( coarse level )

        % dependent properties
        N_points_axis ( 1, : ) { mustBePositive, mustBeInteger }	% number of points along each axis
        qmf ( 1, : ) double                                         % quadrature mirror filter
        handle_fwd                                                  % function handle to forward transform
        handle_inv                                                  % function handle to inverse transform

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = wavelet( strs_type, parameters, N_dimensions, scales_finest, scales_coarsest )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array for strs_type
            if ~iscell( strs_type )
                strs_type = { strs_type };
            end

% TODO: check number of dimensions

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( strs_type, parameters, N_dimensions, scales_finest, scales_coarsest );

            %--------------------------------------------------------------
            % 2.) create discrete wavelet transforms
            %--------------------------------------------------------------
            % total number of lattice points
            N_points = ( 2.^scales_finest ).^N_dimensions;

            % constructor of superclass
            objects@linear_transforms.orthonormal_linear_transform( N_points );

            % iterate discrete wavelet transforms
            for index_object = 1:numel( objects )

                %----------------------------------------------------------
                % a) set independent properties
                %----------------------------------------------------------
                objects( index_object ).type = strs_type{ index_object };
                objects( index_object ).parameter = parameters( index_object );
                objects( index_object ).N_dimensions = N_dimensions( index_object );
                objects( index_object ).scale_finest = scales_finest( index_object );
                objects( index_object ).scale_coarsest = scales_coarsest( index_object );

                %----------------------------------------------------------
                % b) set dependent properties
                %----------------------------------------------------------
                % number of points along each axis
                objects( index_object ).N_points_axis = repmat( 2.^objects( index_object ).scale_finest, [ 1, objects( index_object ).N_dimensions ] );

                % compute quadrature mirror filter (QMF)
                objects( index_object ).qmf = MakeONFilter( objects( index_object ).type, objects( index_object ).parameter );

                % specify transform functions
                switch objects( index_object ).N_dimensions

                    case 1

                        %--------------------------------------------------
                        % one-dimensional transform
                        %--------------------------------------------------
                        objects( index_object ).handle_fwd = @( x ) FWT_PO( x, objects( index_object ).scale_coarsest, objects( index_object ).qmf );
                        objects( index_object ).handle_inv = @( x ) IWT_PO( x, objects( index_object ).scale_coarsest, objects( index_object ).qmf );

                    case 2

                        %--------------------------------------------------
                        % two-dimensional transforms
                        %--------------------------------------------------
                        objects( index_object ).handle_fwd = @( x ) FWT2_PO( x, objects( index_object ).scale_coarsest, objects( index_object ).qmf );
                        objects( index_object ).handle_inv = @( x ) IWT2_PO( x, objects( index_object ).scale_coarsest, objects( index_object ).qmf );

                end % switch objects( index_object ).N_dimensions

            end % for index_object = 1:numel( objects )

        end % function objects = wavelet( strs_type, parameters, N_dimensions, scales_finest, scales_coarsest )

        %------------------------------------------------------------------
        % forward transform (overload forward_transform method)
        %------------------------------------------------------------------
        function y = forward_transform( LTs, x )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array for x
            if ~iscell( x )
                x = { x };
            end

            % multiple LTs / single x
            if ~isscalar( LTs ) && isscalar( x )
                x = repmat( x, size( LTs ) );
            end

            % single LTs / multiple x
            if isscalar( LTs ) && ~isscalar( x )
                x = repmat( LTs, size( x ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( LTs, x );

            %--------------------------------------------------------------
            % 2.) compute forward wavelet transform
            %--------------------------------------------------------------
            % specify cell array for y
            y = cell( size( LTs ) );

            % iterate discrete wavelet transforms
            for index_object = 1:numel( LTs )

                % ensure numeric matrix
                if ~( isnumeric( x{ index_object } ) && ismatrix( x{ index_object } ) )
                    errorStruct.message = sprintf( 'x{ %d } must be a numeric matrix!', index_object );
                    errorStruct.identifier = 'forward_transform:NoNumericMatrix';
                    error( errorStruct );
                end

                % number of vectors to transform
                N_signals = size( x{ index_object }, 2 );

                % initialize results with zeros
                y{ index_object } = zeros( size( x{ index_object } ) );

                % iterate signals
                for index_signal = 1:N_signals

                    % prepare shape of matrix
                    x_act = reshape( x{ index_object }( :, index_signal ), LTs( index_object ).N_points_axis );

                    % apply forward transform
                    y_act = LTs( index_object ).handle_fwd( real( x_act ) );
                    y_act = y_act + 1j * LTs( index_object ).handle_fwd( imag( x_act ) );

                    % save result as column vector
                    y{ index_object }( :, index_signal ) = y_act( : );

                end % for index_signal = 1:N_signals

            end % for index_object = 1:numel( LTs )

            % avoid cell array for single diagonal weighting matrix
            if isscalar( LTs )
                y = y{ 1 };
            end

        end % function y = forward_transform( LTs, x )

        %------------------------------------------------------------------
        % adjoint transform (overload adjoint_transform method)
        %------------------------------------------------------------------
        function y = adjoint_transform( LTs, x )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array for x
            if ~iscell( x )
                x = { x };
            end

            % multiple LTs / single x
            if ~isscalar( LTs ) && isscalar( x )
                x = repmat( x, size( LTs ) );
            end

            % single LTs / multiple x
            if isscalar( LTs ) && ~isscalar( x )
                x = repmat( LTs, size( x ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( LTs, x );

            %--------------------------------------------------------------
            % 2.) compute forward wavelet transform
            %--------------------------------------------------------------
            % specify cell array for y
            y = cell( size( LTs ) );

            % iterate discrete wavelet transforms
            for index_object = 1:numel( LTs )

                % ensure numeric matrix
                if ~( isnumeric( x{ index_object } ) && ismatrix( x{ index_object } ) )
                    errorStruct.message = sprintf( 'x{ %d } must be a numeric matrix!', index_object );
                    errorStruct.identifier = 'forward_transform:NoNumericMatrix';
                    error( errorStruct );
                end

                % number of vectors to transform
                N_signals = size( x{ index_object }, 2 );

                % initialize results with zeros
                y{ index_object } = zeros( size( x{ index_object } ) );

                % iterate signals
                for index_signal = 1:N_signals

                    % prepare shape of matrix
                    x_act = reshape( x{ index_object }( :, index_signal ), LTs( index_object ).N_points_axis );

                    % apply inverse transform
                    y_act = LTs( index_object ).handle_inv( real( x_act ) );
                    y_act = y_act + 1j * LTs( index_object ).handle_inv( imag( x_act ) );

                    % save result as column vector
                    y{ index_object }( :, index_signal ) = y_act( : );

                end % for index_signal = 1:N_signals

            end % for index_object = 1:numel( LTs )

            % avoid cell array for single diagonal weighting matrix
            if isscalar( LTs )
                y = y{ 1 };
            end

        end % function y = adjoint_transform( LTs, x )

    end % methods

end % classdef wavelet < linear_transforms.orthonormal_linear_transform
