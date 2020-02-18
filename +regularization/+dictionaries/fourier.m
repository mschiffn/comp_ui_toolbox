%
% superclass for all Fourier dictionaries
%
% author: Martin F. Schiffner
% date: 2019-12-28
% modified: 2020-02-17
%
classdef fourier < regularization.dictionaries.dictionary

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = fourier( varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure nonempty size
            if nargin >= 1 && ~isempty( varargin{ 1 } )
                size = varargin{ 1 };
            else
                size = 1;
            end

            % superclass ensures row vector for size
            % superclass ensures nonempty positive integers for size

            %--------------------------------------------------------------
            % 2.) create Fourier dictionaries
            %--------------------------------------------------------------
            % constructor of superclass
            objects@regularization.dictionaries.dictionary( size );

        end % function objects = fourier( varargin )

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        function strs_out = string( dictionaries_fourier )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class regularization.dictionaries.fourier
            if ~isa( dictionaries_fourier, 'regularization.dictionaries.fourier' )
                errorStruct.message = 'dictionaries_fourier must be regularization.dictionaries.fourier!';
                errorStruct.identifier = 'string:NoOptionsDictionaryFourier';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create string array
            %--------------------------------------------------------------
            % repeat string "Fourier"
            strs_out = repmat( "Fourier", size( dictionaries_fourier ) );

        end % function strs_out = string( dictionaries_fourier )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected, hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % create linear transform (scalar)
        %------------------------------------------------------------------
        function LT = get_LT_scalar( ~, operator_born )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class regularization.dictionaries.dictionary (scalar)
            % calling function ensures class scattering.operator_born (scalar)

            %--------------------------------------------------------------
            % 2.) create linear transform (scalar)
            %--------------------------------------------------------------
            % create linear transform
            LT = linear_transforms.fourier( operator_born.sequence.setup.FOV.shape.grid.N_points_axis );

        end % function LT = get_LT_scalar( ~, operator_born )

	end % methods (Access = protected, Hidden)

end % classdef fourier < regularization.dictionaries.dictionary
