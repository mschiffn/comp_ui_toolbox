%
% one- or two-dimensional
% discrete wavelet transforms for
% various options
% (requires WaveLab: http://www-stat.stanford.edu/~wavelab)
%
% author: Martin F. Schiffner
% date: 2016-08-13
% modified: 2020-01-30
%
classdef wavelet < linear_transforms.linear_transform_vector

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        type ( 1, 1 ) linear_transforms.wavelets.type { mustBeNonempty } = linear_transforms.wavelets.haar	% type of wavelet
        N_dimensions ( 1, 1 ) { mustBePositive, mustBeInteger, mustBeNonempty } = 2         % number of dimensions
        scale_finest ( 1, 1 ) { mustBeNonnegative, mustBeInteger, mustBeNonempty } = 9      % finest scale ( fine level )
        scale_coarsest ( 1, 1 ) { mustBeNonnegative, mustBeInteger, mustBeNonempty } = 0	% coarsest scale ( coarse level )

        % dependent properties
        N_points_axis ( 1, : ) { mustBePositive, mustBeInteger, mustBeNonempty } = [ 512, 512 ]	% number of points along each axis ( dyadic )
        qmf ( 1, : ) double                                         % quadrature mirror filter
        handle_fwd ( 1, 1 ) function_handle { mustBeNonempty } = @( x ) x	% function handle to forward transform
        handle_inv ( 1, 1 ) function_handle { mustBeNonempty } = @( x ) x	% function handle to inverse transform

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = wavelet( types, N_dimensions, scales_finest, scales_coarsest )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.wavelets.type
            if ~isa( types, 'linear_transforms.wavelets.type' )
                errorStruct.message = 'types must be linear_transforms.wavelets.type!';
                errorStruct.identifier = 'wavelet:NoWaveletTypes';
                error( errorStruct );
            end

            % ensure positive integers for N_dimensions
            mustBeInteger( N_dimensions );
            mustBePositive( N_dimensions );

            % ensure positive integers for scales_finest
            mustBeInteger( scales_finest );
            mustBePositive( scales_finest );

            % ensure nonnegative integers for scales_coarsest
            mustBeInteger( scales_coarsest );
            mustBeNonnegative( scales_coarsest );

            % ensure valid scales
            if any( scales_coarsest( : ) >= scales_finest( : ) )
                errorStruct.message = 'scales_coarsest must be less than scales_finest!';
                errorStruct.identifier = 'wavelet:InvalidCoarsestScales';
                error( errorStruct );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( types, N_dimensions, scales_finest, scales_coarsest );

            %--------------------------------------------------------------
            % 2.) create discrete wavelet transforms
            %--------------------------------------------------------------
            % compute quadrature mirror filters (QMF)
            QMFs = MakeONFilter( types );

            % ensure cell array for QMFs
            if ~iscell( QMFs )
                QMFs = { QMFs };
            end

            % total number of grid points
            N_points = ( 2.^scales_finest ).^N_dimensions;

            % constructor of superclass
            objects@linear_transforms.linear_transform_vector( N_points, N_points );

            % iterate discrete wavelet transforms
            for index_object = 1:numel( objects )

                %----------------------------------------------------------
                % a) set independent properties
                %----------------------------------------------------------
                objects( index_object ).type = types( index_object );
                objects( index_object ).N_dimensions = N_dimensions( index_object );
                objects( index_object ).scale_finest = scales_finest( index_object );
                objects( index_object ).scale_coarsest = scales_coarsest( index_object );

                %----------------------------------------------------------
                % b) set dependent properties
                %----------------------------------------------------------
                % number of points along each axis
                objects( index_object ).N_points_axis = repmat( 2.^objects( index_object ).scale_finest, [ 1, objects( index_object ).N_dimensions ] );

                % quadrature mirror filter (QMF)
                objects( index_object ).qmf = QMFs{ index_object };

                % specify transform functions
                switch objects( index_object ).N_dimensions

                    case 1

                        %--------------------------------------------------
                        % i.) one-dimensional transform
                        %--------------------------------------------------
                        objects( index_object ).handle_fwd = @FWT_PO;
                        objects( index_object ).handle_inv = @IWT_PO;

                    case 2

                        %--------------------------------------------------
                        % ii.) two-dimensional transforms
                        %--------------------------------------------------
                        objects( index_object ).handle_fwd = @FWT2_PO;
                        objects( index_object ).handle_inv = @IWT2_PO;

                    otherwise

                        %--------------------------------------------------
                        % iii.) invalid number of dimensions
                        %--------------------------------------------------
                        errorStruct.message = sprintf( 'objects( %d ).N_dimensions must equal 1 or 2!', index_object );
                        errorStruct.identifier = 'wavelet:InvalidNumberDimensions';
                        error( errorStruct );

                end % switch objects( index_object ).N_dimensions

            end % for index_object = 1:numel( objects )

        end % function objects = wavelet( types, N_dimensions, scales_finest, scales_coarsest )

    end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % forward transform (single vector)
        %------------------------------------------------------------------
        function y = forward_transform_vector( LT, x )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.wavelet (scalar)
            if ~( isa( LT, 'linear_transforms.wavelet' ) && isscalar( LT ) )
                errorStruct.message = 'LT must be linear_transforms.wavelet!';
                errorStruct.identifier = 'forward_transform_vector:NoSingleWaveletTransform';
                error( errorStruct );
            end

            % superclass ensures numeric column vector for x
            % superclass ensures equal numbers of points for x

            %--------------------------------------------------------------
            % 2.) compute forward wavelet transform (single vector)
            %--------------------------------------------------------------
            % prepare shape of vector
            if LT.N_dimensions >= 2
                x = reshape( x, LT.N_points_axis );
            end

            % apply forward transform
            y_act = LT.handle_fwd( real( x ), LT.scale_coarsest, LT.qmf );
            y_act = y_act + 1j * LT.handle_fwd( imag( x ), LT.scale_coarsest, LT.qmf );

            % return result as column vector
            y = y_act( : );

        end % function y = forward_transform_vector( LT, x )

        %------------------------------------------------------------------
        % adjoint transform (single vector)
        %------------------------------------------------------------------
        function y = adjoint_transform_vector( LT, x )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.wavelet (scalar)
            if ~( isa( LT, 'linear_transforms.wavelet' ) && isscalar( LT ) )
                errorStruct.message = 'LT must be linear_transforms.wavelet!';
                errorStruct.identifier = 'adjoint_transform_vector:NoSingleWaveletTransform';
                error( errorStruct );
            end

            % superclass ensures numeric column vector for x
            % superclass ensures equal numbers of coefficients

            %--------------------------------------------------------------
            % 2.) compute adjoint wavelet transform (single vector)
            %--------------------------------------------------------------
            % prepare shape of vector
            if LT.N_dimensions >= 2
                x = reshape( x, LT.N_points_axis );
            end

            % apply inverse transform
            y_act = LT.handle_inv( real( x ), LT.scale_coarsest, LT.qmf );
            y_act = y_act + 1j * LT.handle_inv( imag( x ), LT.scale_coarsest, LT.qmf );

            % return result as column vector
            y = y_act( : );

        end % function y = adjoint_transform_vector( LT, x )

	end % methods (Access = private, Hidden)

end % classdef wavelet < linear_transforms.linear_transform_vector
