%
% superclass for identity dictionary options
%
% author: Martin F. Schiffner
% date: 2019-12-28
% modified: 2020-01-15
%
classdef dictionary_identity < regularization.options.dictionary

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = dictionary_identity( varargin )

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
            % superclass ensures nonempty positive integers

            %--------------------------------------------------------------
            % 2.) create identity dictionary options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@regularization.options.dictionary( size );

        end % function objects = dictionary_identity( varargin )

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        function strs_out = string( dictionaries_identity )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class regularization.options.dictionary_identity
            if ~isa( dictionaries_identity, 'regularization.options.dictionary_identity' )
                errorStruct.message = 'dictionaries_identity must be regularization.options.dictionary_identity!';
                errorStruct.identifier = 'string:NoOptionsDictionaryIdentity';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create string array
            %--------------------------------------------------------------
            % repeat string "identity"
            strs_out = repmat( "identity", size( dictionaries_identity ) );

        end % function strs_out = string( dictionaries_identity )

	end % methods

end % classdef dictionary_identity < regularization.options.dictionary
