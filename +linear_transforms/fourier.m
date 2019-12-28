%
% compute d-dimensional discrete Fourier transform for various options
%
% author: Martin F. Schiffner
% date: 2016-08-12
% modified: 2019-04-29
%
classdef fourier < linear_transforms.orthonormal_linear_transform

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        N_points_axis ( 1, : ) double { mustBePositive, mustBeInteger, mustBeNonempty } = [ 512, 512 ]

        % dependent properties
        N_dimensions ( 1, 1 ) double { mustBePositive, mustBeInteger, mustBeNonempty } = 2
        N_points_sqrt

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = fourier( N_points_axis )
% TODO: vectorize
            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure positive integers
            mustBeInteger( N_points_axis );
            mustBePositive( N_points_axis );

            
            % total number of lattice points
            N_dim = numel( N_points_axis );
            N_points = prod( N_points_axis );

            %--------------------------------------------------------------
            % 2.) create discrete Fourier transforms
            %--------------------------------------------------------------
            % constructor of superclass
            objects@linear_transforms.orthonormal_linear_transform( N_points );

            % internal properties
            objects.N_dimensions = N_dim;
            objects.N_points_axis = N_points_axis;
            objects.N_points_sqrt = sqrt( N_points );

        end % function objects = fourier( N_points_axis )

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
            % 2.) compute forward Fourier transforms
            %--------------------------------------------------------------
            % specify cell array for y
            y = cell( size( LTs ) );

            % iterate discrete Fourier transforms
            for index_object = 1:numel( LTs )

                % ensure numeric matrix
                if ~( isnumeric( x{ index_object } ) && ismatrix( x{ index_object } ) )
                    errorStruct.message = sprintf( 'x{ %d } must be a numeric matrix!', index_object );
                    errorStruct.identifier = 'forward_transform:NoNumericMatrix';
                    error( errorStruct );
                end

% TODO: check compatibility  && isequal( operator_born.discretization.spatial.FOV.shape.grid.N_points_axis, varargin{ 1 }.N_lattice )

                % number of vectors to transform
                N_signals = size( x{ index_object }, 2 );

                % initialize results with zeros
                y{ index_object } = zeros( size( x{ index_object } ) );

                % iterate signals
                for index_signal = 1:N_signals

                    % prepare shape of matrix
                    x_act = reshape( x{ index_object }( :, index_signal ), LTs( index_object ).N_points_axis );

                    % apply forward transform
                    y_act = fftn( x_act ) / LTs( index_object ).N_points_sqrt;

                    % save result as column vector
                    y{ index_object }( :, index_signal ) = y_act( : );

                end % for index_signal = 1:N_signals

            end % for index_object = 1:numel( LTs )

            % avoid cell array for single LTs
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
            % 2.) compute adjoint Fourier transforms
            %--------------------------------------------------------------
            % specify cell array for y
            y = cell( size( LTs ) );

            % iterate discrete Fourier transforms
            for index_object = 1:numel( LTs )

                % ensure numeric matrix
                if ~( isnumeric( x{ index_object } ) && ismatrix( x{ index_object } ) )
                    errorStruct.message = sprintf( 'x{ %d } must be a numeric matrix!', index_object );
                    errorStruct.identifier = 'adjoint_transform:NoNumericMatrix';
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

                    % apply adjoint transform
                    y_act = ifftn( x_act ) * LTs( index_object ).N_points_sqrt;

                    % save result as column vector
                    y{ index_object }( :, index_signal ) = y_act( : );

                end % for index_signal = 1:N_signals

            end % for index_object = 1:numel( LTs )

            % avoid cell array for single LTs
            if isscalar( LTs )
                y = y{ 1 };
            end

        end % function y = adjoint_transform( LTs, x )

    end % methods

end % classdef fourier < linear_transforms.orthonormal_linear_transform
