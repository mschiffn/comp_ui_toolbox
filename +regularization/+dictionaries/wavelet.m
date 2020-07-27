%
% superclass for all wavelet dictionaries
%
% author: Martin F. Schiffner
% date: 2020-01-12
% modified: 2020-07-13
%
classdef wavelet < regularization.dictionaries.dictionary

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties

        % independent properties
        type ( 1, 1 ) linear_transforms.wavelets.type { mustBeNonempty } = linear_transforms.wavelets.symmlet( 10 )	% type of wavelet
        levels ( 1, 1 ) { mustBeNonnegative, mustBeInteger } = 1	% decomposition levels

    end

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = wavelet( types, levels )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure two arguments
            narginchk( 2, 2 );

            % ensure class linear_transforms.wavelets.type
            if ~isa( types, 'linear_transforms.wavelets.type' )
                errorStruct.message = 'types must be linear_transforms.wavelets.type!';
                errorStruct.identifier = 'wavelet:NoWaveletTypes';
                error( errorStruct );
            end

            % ensure nonempty positive integers
            mustBePositive( levels );
            mustBeInteger( levels );
            mustBeNonempty( levels );

            % ensure equal number of dimensions and sizes
            [ types, levels ] = auxiliary.ensureEqualSize( types, levels );

            %--------------------------------------------------------------
            % 2.) create wavelet dictionaries
            %--------------------------------------------------------------
            % constructor of superclass
            objects@regularization.dictionaries.dictionary( size( types ) );

            % iterate wavelet dictionaries
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).type = types( index_object );
                objects( index_object ).levels = levels( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = wavelet( types, levels )

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        function strs_out = string( dictionaries_wavelet )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class regularization.dictionaries.wavelet
            if ~isa( dictionaries_wavelet, 'regularization.dictionaries.wavelet' )
                errorStruct.message = 'dictionaries_wavelet must be regularization.dictionaries.wavelet!';
                errorStruct.identifier = 'string:NoOptionsDictionaryWavelet';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create string array
            %--------------------------------------------------------------
            % repeat string "wavelet"
            strs_out = repmat( "wavelet", size( dictionaries_wavelet ) );

        end % function strs_out = string( dictionaries_wavelet )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected, hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % create linear transform (scalar)
        %------------------------------------------------------------------
        function LT = get_LT_scalar( dictionary, operator_born )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class regularization.dictionaries.dictionary (scalar)
            % calling function ensures class scattering.operator_born (scalar)

            %--------------------------------------------------------------
            % 2.) create linear transform (scalar)
            %--------------------------------------------------------------
            % specify scales
            indicator_dimensions = operator_born.sequence.setup.FOV.shape.grid.N_points_axis > 1;
            N_dimensions = sum( indicator_dimensions );
            scales_finest = log2( operator_born.sequence.setup.FOV.shape.grid.N_points_axis( indicator_dimensions ) );
            scales_coarsest = scales_finest - dictionary.levels;

            % create linear transform
            LT = linear_transforms.wavelet( dictionary.type, N_dimensions, scales_finest( 1 ), scales_coarsest( 1 ) );

        end % function LT = get_LT_scalar( dictionary, operator_born )

	end % methods (Access = protected, Hidden)

end % classdef wavelet < regularization.dictionaries.dictionary
