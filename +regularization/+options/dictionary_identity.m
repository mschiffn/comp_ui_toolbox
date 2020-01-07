%
% superclass for identity dictionary options
%
% author: Martin F. Schiffner
% date: 2019-12-28
% modified: 2020-01-03
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

	end % methods

end % classdef dictionary_identity < regularization.options.dictionary
