%
% superclass for all wavelet dictionary options
%
% author: Martin F. Schiffner
% date: 2020-01-12
% modified: 2020-01-28
%
classdef dictionary_wavelet < regularization.options.dictionary

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
        function objects = dictionary_wavelet( types, levels )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.wavelets.type
            if ~isa( types, 'linear_transforms.wavelets.type' )
                errorStruct.message = 'types must be linear_transforms.wavelets.type!';
                errorStruct.identifier = 'dictionary_wavelet:NoWaveletTypes';
                error( errorStruct );
            end

            % ensure nonempty positive integers
            mustBePositive( levels );
            mustBeInteger( levels );
            mustBeNonempty( levels );

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( types, levels );

            %--------------------------------------------------------------
            % 2.) create wavelet dictionary options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@regularization.options.dictionary( size( types ) );

            % iterate wavelet dictionary options
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).type = types( index_object );
                objects( index_object ).levels = levels( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = dictionary_wavelet( types, levels )

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        function strs_out = string( dictionaries_wavelet )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class regularization.options.dictionary_wavelet
            if ~isa( dictionaries_wavelet, 'regularization.options.dictionary_wavelet' )
                errorStruct.message = 'dictionaries_wavelet must be regularization.options.dictionary_wavelet!';
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

end % classdef dictionary_wavelet < regularization.options.dictionary
